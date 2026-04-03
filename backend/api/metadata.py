"""
Metadata API Endpoints
Handles patient metadata storage, analysis results, and treatment outcomes
"""

from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form
from sqlalchemy.orm import Session
from typing import List, Optional
from datetime import datetime
import json

from backend.database_init import get_db
from backend.models import (
    PatientMetadata,
    AnalysisResult,
    TreatmentOutcome,
    TumorDrugInteraction,
    ModelTrainingHistory,
    PerformanceMetric,
    Patient
)

router = APIRouter(prefix="/api/metadata", tags=["metadata"])


# ============================================================================
# PATIENT METADATA ENDPOINTS
# ============================================================================

@router.post("/patients/{patient_id}/analyses")
async def create_patient_analysis(
    patient_id: int,
    ct_scan_metadata: str = Form(...),  # JSON string
    physician_notes: str = Form(...),
    db: Session = Depends(get_db)
):
    """
    Create new analysis for patient
    Stores metadata and triggers analysis
    """
    # Verify patient exists
    patient = db.query(Patient).filter(Patient.id == patient_id).first()
    if not patient:
        raise HTTPException(status_code=404, detail="Patient not found")
    
    # Determine version number
    last_metadata = db.query(PatientMetadata).filter(
        PatientMetadata.patient_id == patient_id
    ).order_by(PatientMetadata.version.desc()).first()
    
    new_version = (last_metadata.version + 1) if last_metadata else 1
    
    # Parse metadata
    try:
        ct_metadata = json.loads(ct_scan_metadata) if ct_scan_metadata else {}
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid CT metadata JSON")
    
    # Create patient metadata
    metadata = PatientMetadata(
        patient_id=patient_id,
        version=new_version,
        analysis_date=datetime.now(),
        ct_scan_metadata=ct_metadata,
        physician_notes=physician_notes,
        # Features will be populated by extraction pipeline
        tumor_features=None,
        cell_features=None,
        clinical_features=None,
        combined_metadata_vector=None
    )
    
    db.add(metadata)
    db.commit()
    db.refresh(metadata)
    
    return {
        "success": True,
        "metadata_id": metadata.id,
        "patient_id": patient_id,
        "version": new_version,
        "message": "Patient metadata created. Ready for feature extraction."
    }


@router.get("/patients/{patient_id}/analyses")
async def get_patient_analyses(
    patient_id: int,
    db: Session = Depends(get_db)
):
    """
    Get all analysis versions for patient
    """
    metadata_list = db.query(PatientMetadata).filter(
        PatientMetadata.patient_id == patient_id
    ).order_by(PatientMetadata.version.desc()).all()
    
    if not metadata_list:
        raise HTTPException(status_code=404, detail="No analyses found for this patient")
    
    return {
        "patient_id": patient_id,
        "total_analyses": len(metadata_list),
        "analyses": [m.to_dict() for m in metadata_list]
    }


@router.get("/patients/{patient_id}/analyses/latest")
async def get_latest_analysis(
    patient_id: int,
    db: Session = Depends(get_db)
):
    """
    Get most recent analysis for patient
    """
    metadata = db.query(PatientMetadata).filter(
        PatientMetadata.patient_id == patient_id
    ).order_by(PatientMetadata.version.desc()).first()
    
    if not metadata:
        raise HTTPException(status_code=404, detail="No analyses found for this patient")
    
    # Get associated analysis results
    results = db.query(AnalysisResult).filter(
        AnalysisResult.patient_metadata_id == metadata.id
    ).order_by(AnalysisResult.created_at.desc()).first()
    
    return {
        "metadata": metadata.to_dict(),
        "results": results.to_dict() if results else None
    }


@router.put("/patients/{patient_id}/analyses/{version}/features")
async def update_extracted_features(
    patient_id: int,
    version: int,
    tumor_features: dict,
    cell_features: Optional[dict] = None,
    clinical_features: Optional[dict] = None,
    combined_vector: Optional[List[float]] = None,
    db: Session = Depends(get_db)
):
    """
    Update metadata with extracted features
    Called by metadata extraction pipeline
    """
    metadata = db.query(PatientMetadata).filter(
        PatientMetadata.patient_id == patient_id,
        PatientMetadata.version == version
    ).first()
    
    if not metadata:
        raise HTTPException(status_code=404, detail="Metadata not found")
    
    # Update features
    metadata.tumor_features = tumor_features
    if cell_features:
        metadata.cell_features = cell_features
    if clinical_features:
        metadata.clinical_features = clinical_features
    if combined_vector:
        metadata.combined_metadata_vector = combined_vector
    
    db.commit()
    
    return {
        "success": True,
        "message": "Features updated successfully"
    }


# ============================================================================
# ANALYSIS RESULTS ENDPOINTS
# ============================================================================

@router.post("/results")
async def create_analysis_result(
    patient_metadata_id: int,
    tumors: dict,
    drug_recommendations: dict,
    confidence_scores: dict,
    model_version: str,
    db: Session = Depends(get_db)
):
    """
    Store analysis results
    """
    result = AnalysisResult(
        patient_metadata_id=patient_metadata_id,
        tumors=tumors,
        drug_recommendations=drug_recommendations,
        confidence_scores=confidence_scores,
        risk_assessment=None,  # Can be added later
        model_version=model_version,
        model_parameters=None,
        doctor_accepted=None,
        doctor_notes=None
    )
    
    db.add(result)
    db.commit()
    db.refresh(result)
    
    return {
        "success": True,
        "result_id": result.id,
        "message": "Analysis result saved"
    }


@router.put("/results/{result_id}/feedback")
async def submit_doctor_feedback(
    result_id: int,
    accepted: bool,
    notes: Optional[str] = None,
    db: Session = Depends(get_db)
):
    """
    Doctor feedback on analysis result
    CRITICAL for learning loop
    """
    result = db.query(AnalysisResult).filter(AnalysisResult.id == result_id).first()
    
    if not result:
        raise HTTPException(status_code=404, detail="Result not found")
    
    result.doctor_accepted = accepted
    result.doctor_notes = notes
    result.doctor_feedback_date = datetime.now()
    
    db.commit()
    
    return {
        "success": True,
        "message": "Feedback recorded. Thank you!"
    }


# ============================================================================
# TREATMENT OUTCOMES ENDPOINTS
# ============================================================================

@router.post("/treatments/outcomes")
async def create_treatment_outcome(
    patient_id: int,
    analysis_result_id: int,
    prescribed_cocktail: dict,
    baseline_tumor_size: float,
    baseline_tumor_count: int,
    predicted_efficacy: float,
    db: Session = Depends(get_db)
):
    """
    Record treatment initiation
    Will be updated later with actual outcomes
    """
    outcome = TreatmentOutcome(
        patient_id=patient_id,
        analysis_result_id=analysis_result_id,
        prescribed_cocktail=prescribed_cocktail,
        treatment_start_date=datetime.now().date(),
        baseline_tumor_size=baseline_tumor_size,
        baseline_tumor_count=baseline_tumor_count,
        predicted_efficacy=predicted_efficacy,
        # Actual outcomes will be added during follow-up
        tumor_response=None,
        actual_efficacy=None
    )
    
    db.add(outcome)
    db.commit()
    db.refresh(outcome)
    
    return {
        "success": True,
        "outcome_id": outcome.id,
        "message": "Treatment outcome tracking initiated"
    }


@router.put("/treatments/outcomes/{outcome_id}")
async def update_treatment_outcome(
    outcome_id: int,
    tumor_response: str,
    tumor_size_change_percent: float,
    tumor_count_change: int,
    side_effects: Optional[dict] = None,
    quality_of_life_score: Optional[float] = None,
    survival_months: Optional[int] = None,
    db: Session = Depends(get_db)
):
    """
    Update treatment outcome with actual results
    This is the LEARNING DATA that drives improvement
    """
    outcome = db.query(TreatmentOutcome).filter(TreatmentOutcome.id == outcome_id).first()
    
    if not outcome:
        raise HTTPException(status_code=404, detail="Outcome not found")
    
    # Update actual results
    outcome.tumor_response = tumor_response
    outcome.tumor_size_change_percent = tumor_size_change_percent
    outcome.tumor_count_change = tumor_count_change
    outcome.follow_up_date = datetime.now().date()
    
    if side_effects:
        outcome.side_effects = side_effects
    if quality_of_life_score:
        outcome.quality_of_life_score = quality_of_life_score
    if survival_months:
        outcome.survival_months = survival_months
    
    # Calculate actual efficacy based on tumor response
    if tumor_response == "complete_response":
        actual_efficacy = 1.0
    elif tumor_response == "partial_response":
        # Based on size reduction percentage
        actual_efficacy = min(abs(tumor_size_change_percent) / 100, 0.9)
    elif tumor_response == "stable":
        actual_efficacy = 0.5
    else:  # progressive
        actual_efficacy = 0.2
    
    outcome.actual_efficacy = actual_efficacy
    
    # Calculate prediction error
    if outcome.predicted_efficacy:
        outcome.prediction_error = abs(outcome.predicted_efficacy - actual_efficacy)
    
    db.commit()
    
    return {
        "success": True,
        "predicted_efficacy": outcome.predicted_efficacy,
        "actual_efficacy": actual_efficacy,
        "prediction_error": outcome.prediction_error,
        "message": "Treatment outcome updated. Data ready for learning!"
    }


# ============================================================================
# TUMOR-DRUG INTERACTION ENDPOINTS
# ============================================================================

@router.get("/interactions")
async def get_interactions(
    tumor_type: Optional[str] = None,
    min_evidence_count: int = 5,
    db: Session = Depends(get_db)
):
    """
    Get learned tumor-drug interactions
    """
    query = db.query(TumorDrugInteraction).filter(
        TumorDrugInteraction.evidence_count >= min_evidence_count
    )
    
    if tumor_type:
        query = query.filter(TumorDrugInteraction.tumor_type == tumor_type)
    
    interactions = query.order_by(TumorDrugInteraction.efficacy_score.desc()).limit(50).all()
    
    return {
        "total": len(interactions),
        "interactions": [i.to_dict() for i in interactions]
    }


# ============================================================================
# PERFORMANCE METRICS ENDPOINTS
# ============================================================================

@router.get("/performance")
async def get_performance_metrics(
    days: int = 30,
    db: Session = Depends(get_db)
):
    """
    Get performance metrics for dashboard
    """
    from datetime import date, timedelta
    
    start_date = date.today() - timedelta(days=days)
    
    metrics = db.query(PerformanceMetric).filter(
        PerformanceMetric.metric_date >= start_date
    ).order_by(PerformanceMetric.metric_date.desc()).all()
    
    return {
        "period_days": days,
        "metrics": [m.to_dict() for m in metrics]
    }


@router.get("/performance/latest")
async def get_latest_metrics(db: Session = Depends(get_db)):
    """
    Get most recent performance metrics
    """
    metric = db.query(PerformanceMetric).order_by(
        PerformanceMetric.metric_date.desc()
    ).first()
    
    if not metric:
        return {
            "message": "No metrics available yet"
        }
    
    return metric.to_dict()
