"""
ADDS Inference API
Pathway-based drug recommendations
"""

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from datetime import datetime
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from database import get_db
from database.models import CTAnalysis
from services.adds_service import ADDSService

router = APIRouter()
adds_service = ADDSService()


@router.post("/analysis/{analysis_id}/infer")
async def run_adds_inference(
    analysis_id: int,
    db: Session = Depends(get_db)
):
    """
    Run ADDS pathway-based inference
    
    Args:
        analysis_id: CT Analysis ID
        
    Returns:
        ADDS drug recommendations and rationale
    """
    # Get analysis
    analysis = db.query(CTAnalysis).filter(CTAnalysis.id == analysis_id).first()
    if not analysis:
        raise HTTPException(status_code=404, detail="Analysis not found")
    
    # Check if CT analysis is complete
    if analysis.status != "completed":
        raise HTTPException(
            status_code=400,
            detail=f"CT analysis must be completed first. Current status: {analysis.status}"
        )
    
    # Check required data
    if not analysis.radiomics_features or not analysis.tumor_characteristics:
        raise HTTPException(
            status_code=400,
            detail="CT analysis must include radiomics features and tumor characteristics"
        )
    
    # Prepare clinical data
    clinical_data = {
        "patient_id": analysis.patient.patient_id,
        "tumor_location": analysis.tumor_location or "Unknown",
        "tnm_stage": analysis.tnm_stage or "Unknown",
        "msi_status": analysis.msi_status or "Unknown",
        "kras_mutation": analysis.kras_mutation or "Unknown"
    }
    
    # Run ADDS inference
    try:
        result = adds_service.run_inference(
            radiomics=analysis.radiomics_features,
            tumor_characteristics=analysis.tumor_characteristics,
            clinical_data=clinical_data
        )
        
        # Save result
        analysis.adds_result = result
        db.commit()
        
        return {
            "analysis_id": analysis_id,
            "status": "completed",
            "result": result
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"ADDS inference failed: {str(e)}")


@router.get("/analysis/{analysis_id}/result")
async def get_adds_result(
    analysis_id: int,
    db: Session = Depends(get_db)
):
    """
    Get ADDS inference result
    
    Args:
        analysis_id: CT Analysis ID
        
    Returns:
        ADDS inference result
    """
    analysis = db.query(CTAnalysis).filter(CTAnalysis.id == analysis_id).first()
    if not analysis:
        raise HTTPException(status_code=404, detail="Analysis not found")
    
    if not analysis.adds_result:
        raise HTTPException(
            status_code=404,
            detail="ADDS inference has not been run for this analysis"
        )
    
    return analysis.adds_result
